function _serialize (o)
	str = ""
	if type(o) == "number" then
		--io.write(o)
		str = str .. tostring(o)
	elseif type(o) == "string" then
		--io.write(string.format("%q", o))
		str = str .. string.format("%q", o)
	elseif type(o) == "table" then
		--io.write("{\\n")
		str = str .. "{\n"
		for k,v in pairs(o) do
			--io.write("  ", k, " = ")
			str = str .. " " .. tostring(k) .. " = "
			str = str .. serialize(v, str)
			--io.write(",\n")
			str = str .. ",\n"
		end
		--io.write("}\n")
		str = str .. "}\n"
	else
		error("cannot serialize a " .. type(o))
	end
	return str
end

function serialize(o)
	str = ""
	str = _serialize(o, str)
	return str
end

